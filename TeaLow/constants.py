"""Constants used throughout the TeaLow SDK.

This module centralizes provider identification, default API endpoints,
timeouts, retry configuration, and other static configuration values.
Keeping these values in one place makes it easy to update endpoints or
tune retry/timeout behaviour without touching business logic elsewhere
in the codebase.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple


class Provider(str, Enum):
    """Enumeration of the AI providers supported by TeaLow."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"


#: Ordered mapping of (substring, Provider) used for automatic provider
#: detection based on a model name. Order matters: more specific
#: substrings should be listed before more generic ones.
MODEL_PREFIX_ROUTES: Tuple[Tuple[str, Provider], ...] = (
    ("gemini", Provider.GEMINI),
    ("gpt-", Provider.OPENAI),
    ("o1", Provider.OPENAI),
    ("o3", Provider.OPENAI),
    ("o4", Provider.OPENAI),
    ("text-embedding", Provider.OPENAI),
    ("dall-e", Provider.OPENAI),
    ("claude", Provider.ANTHROPIC),
    ("deepseek", Provider.DEEPSEEK),
)

#: Default base URLs for each provider's REST API.
DEFAULT_BASE_URLS: Dict[Provider, str] = {
    Provider.GEMINI: "https://generativelanguage.googleapis.com/v1beta",
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.ANTHROPIC: "https://api.anthropic.com/v1",
    Provider.DEEPSEEK: "https://api.deepseek.com/v1",
}

#: Environment variable names used to automatically discover API keys
#: when one is not explicitly supplied to the client constructor.
ENV_VAR_NAMES: Dict[Provider, str] = {
    Provider.GEMINI: "GEMINI_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
}

#: Default network timeout (seconds) applied to every outbound request.
DEFAULT_TIMEOUT: float = 60.0

#: Default number of retry attempts for transient failures.
DEFAULT_MAX_RETRIES: int = 3

#: Base delay (seconds) used for exponential backoff between retries.
DEFAULT_BACKOFF_BASE: float = 0.5

#: Maximum delay (seconds) allowed between retries regardless of the
#: exponential backoff calculation.
DEFAULT_BACKOFF_MAX: float = 20.0

#: HTTP status codes that are considered retryable (transient errors).
RETRYABLE_STATUS_CODES: Tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504)

#: HTTP status code specifically reserved for rate limiting.
RATE_LIMIT_STATUS_CODE: int = 429

#: Default sampling temperature used when the caller does not specify one.
DEFAULT_TEMPERATURE: float = 0.7

#: Default maximum number of tokens requested from the provider.
DEFAULT_MAX_TOKENS: int = 1024

#: Current SDK version. Kept in sync with pyproject.toml.
SDK_VERSION: str = "1.0.0"

#: User-Agent header sent with every request.
USER_AGENT: str = f"TeaLow-SDK/{SDK_VERSION} (+https://github.com/tealow/tealow)"
