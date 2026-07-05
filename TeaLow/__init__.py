"""TeaLow: a unified AI SDK for Google Gemini, OpenAI, Anthropic, and DeepSeek.

TeaLow exposes a single, consistent API across multiple AI providers so
that switching providers is as simple as changing a model name::

    from TeaLow import TeaLow

    ai = TeaLow(model="gemini-2.5-flash", api="YOUR_API_KEY")
    response = ai.send("Hello!")
    print(response)

An asynchronous client is available as :class:`AsyncTeaLow` for use in
``asyncio``-based applications.
"""

from __future__ import annotations

from .async_client import AsyncTeaLow
from .client import TeaLow, detect_provider
from .constants import Provider, SDK_VERSION
from .exceptions import (
    APIResponseError,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    FileUploadError,
    ImageGenerationError,
    InvalidAPIKeyError,
    InvalidResponseError,
    JSONModeError,
    MissingAPIKeyError,
    RateLimitError,
    RetryExhaustedError,
    StreamingError,
    TeaLowError,
    TimeoutError,
    UnsupportedModelError,
    VisionInputError,
)
from .models import Conversation, Message, Response, StreamChunk, Usage
from .utils import configure_logging

__version__ = SDK_VERSION

__all__ = [
    "TeaLow",
    "AsyncTeaLow",
    "detect_provider",
    "Provider",
    "Conversation",
    "Message",
    "Response",
    "StreamChunk",
    "Usage",
    "configure_logging",
    "TeaLowError",
    "ConfigurationError",
    "UnsupportedModelError",
    "MissingAPIKeyError",
    "InvalidAPIKeyError",
    "AuthenticationError",
    "RateLimitError",
    "TimeoutError",
    "ConnectionError",
    "APIResponseError",
    "InvalidResponseError",
    "RetryExhaustedError",
    "StreamingError",
    "FileUploadError",
    "ImageGenerationError",
    "VisionInputError",
    "JSONModeError",
    "__version__",
]
