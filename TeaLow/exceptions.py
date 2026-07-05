"""Exception hierarchy for the TeaLow SDK.

All exceptions raised by TeaLow inherit from :class:`TeaLowError`, which
makes it possible for consumers to catch every SDK-originated error with
a single ``except TeaLowError`` clause, while still allowing more precise
handling of specific failure modes (authentication, rate limiting,
timeouts, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class TeaLowError(Exception):
    """Base class for all exceptions raised by the TeaLow SDK."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:  # pragma: no cover - trivial
        parts = [self.message]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the exception to a dictionary, useful for logging."""
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "provider": self.provider,
            "status_code": self.status_code,
            "response_body": self.response_body,
        }


class ConfigurationError(TeaLowError):
    """Raised when the SDK is configured incorrectly (bad model, etc.)."""


class UnsupportedModelError(ConfigurationError):
    """Raised when a model name cannot be mapped to any known provider."""


class MissingAPIKeyError(ConfigurationError):
    """Raised when no API key is supplied and none can be found via env vars."""


class InvalidAPIKeyError(TeaLowError):
    """Raised when the provider rejects the supplied API key (HTTP 401/403)."""


class AuthenticationError(InvalidAPIKeyError):
    """Alias retained for semantic clarity when authentication fails."""


class RateLimitError(TeaLowError):
    """Raised when the provider signals that the caller has been rate limited."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            status_code=status_code,
            response_body=response_body,
        )
        self.retry_after = retry_after


class TimeoutError(TeaLowError):
    """Raised when a request exceeds the configured timeout."""


class ConnectionError(TeaLowError):
    """Raised when a network-level connection failure occurs."""


class APIResponseError(TeaLowError):
    """Raised when the provider returns a non-success HTTP response."""


class InvalidResponseError(TeaLowError):
    """Raised when the provider response cannot be parsed as expected."""


class RetryExhaustedError(TeaLowError):
    """Raised when all configured retry attempts have been exhausted."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        last_exception: Optional[BaseException] = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message, provider=provider)
        self.last_exception = last_exception
        self.attempts = attempts


class StreamingError(TeaLowError):
    """Raised when an error occurs while consuming a streamed response."""


class FileUploadError(TeaLowError):
    """Raised when a file upload to a provider fails."""


class ImageGenerationError(TeaLowError):
    """Raised when image generation fails or is unsupported for a model."""


class VisionInputError(TeaLowError):
    """Raised when an image input is malformed or unsupported for vision."""


class JSONModeError(TeaLowError):
    """Raised when JSON response mode is requested but parsing fails."""
