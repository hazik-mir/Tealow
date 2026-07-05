"""Synchronous client for the TeaLow unified AI SDK.

This module exposes the :class:`TeaLow` class, the primary entry point
for most users of the SDK::

    from TeaLow import TeaLow

    ai = TeaLow(model="gemini-2.5-flash", api="YOUR_API_KEY")
    response = ai.send("Hello!")
    print(response)

The same class works identically across all supported providers; only
the ``model`` name (and the corresponding API key) needs to change.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from .constants import (
    DEFAULT_BASE_URLS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    Provider,
)
from .exceptions import (
    RateLimitError,
    RetryExhaustedError,
    TeaLowError,
    UnsupportedModelError,
)
from .models import Conversation, Response, StreamChunk
from .providers import get_provider_class
from .providers.base import BaseProvider
from .session import AsyncHTTPSession, HTTPSession
from .streaming import StreamAccumulator
from .types import FilePayload, ImagePayload
from .utils import (
    detect_mime_type,
    encode_file_to_base64,
    get_logger,
    is_retryable_status,
    resolve_api_key,
    retry_sync,
    validate_api_key_format,
)


def detect_provider(model: str) -> Provider:
    """Determine which :class:`Provider` a given model name belongs to.

    Args:
        model: The model identifier, e.g. ``"gpt-4o"`` or ``"claude-opus-4-8"``.

    Raises:
        UnsupportedModelError: If the model name does not match any
            known provider prefix.
    """
    from .constants import MODEL_PREFIX_ROUTES

    normalized = model.lower()
    for substring, provider in MODEL_PREFIX_ROUTES:
        if substring in normalized:
            return provider
    raise UnsupportedModelError(
        f"Could not determine a provider for model '{model}'. Supported "
        "providers are: gemini, openai (gpt-/o1/o3/o4), anthropic (claude), "
        "and deepseek."
    )


class TeaLow:
    """Unified synchronous client for interacting with multiple AI providers.

    Args:
        model: Model identifier (e.g. ``"gpt-4o"``, ``"gemini-2.5-flash"``,
            ``"claude-sonnet-4-6"``, ``"deepseek-chat"``). The provider is
            automatically detected from this string.
        api: API key for the detected provider. If omitted, TeaLow will
            look for a provider-specific environment variable (e.g.
            ``OPENAI_API_KEY``).
        system_prompt: Optional system prompt applied to the conversation.
        temperature: Default sampling temperature for generated responses.
        max_tokens: Default maximum number of tokens to generate.
        timeout: Network timeout in seconds applied to every request.
        max_retries: Maximum number of retry attempts for transient failures.
        base_url: Override the default API base URL for the provider.
        proxies: Optional proxy configuration passed to the HTTP session.
        headers: Optional custom headers merged into every request.
        verify_ssl: Whether to verify TLS certificates.
        validate_key: Whether to perform a lightweight format check on
            the API key before making any network calls.

    Example:
        >>> ai = TeaLow(model="gemini-2.5-flash", api="YOUR_API_KEY")
        >>> response = ai.send("Hello!")
        >>> print(response)
    """

    def __init__(
        self,
        model: str,
        api: Optional[str] = None,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_url: Optional[str] = None,
        proxies: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
        validate_key: bool = True,
    ) -> None:
        self.model = model
        self.provider_enum = detect_provider(model)
        self.api_key = resolve_api_key(self.provider_enum, api)
        if validate_key:
            validate_api_key_format(self.provider_enum, self.api_key)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url or DEFAULT_BASE_URLS[self.provider_enum]

        self._logger = get_logger("client")
        self._session = HTTPSession(
            timeout=timeout,
            proxies=proxies,
            custom_headers=headers,
            verify_ssl=verify_ssl,
        )
        self._async_session: Optional[AsyncHTTPSession] = None

        provider_class = get_provider_class(self.provider_enum)
        self._provider: BaseProvider = provider_class(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            session=self._session,
            async_session=None,
            extra_headers=headers,
        )
        self._provider.validate_model()

        self.conversation = Conversation(system_prompt=system_prompt)

    @property
    def provider(self) -> str:
        """The machine-readable name of the detected provider (e.g. ``"openai"``)."""
        return self.provider_enum.value

    def set_system_prompt(self, prompt: Optional[str]) -> None:
        """Update the system prompt used for subsequent messages."""
        self.conversation.set_system_prompt(prompt)

    def reset_history(self) -> None:
        """Clear the conversation history while keeping the system prompt."""
        self.conversation.clear()

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return the full conversation history as plain dictionaries."""
        return self.conversation.as_list()

    @staticmethod
    def load_image(path: Optional[str] = None, *, url: Optional[str] = None) -> ImagePayload:
        """Build an :class:`ImagePayload` from a local file path or a URL.

        Args:
            path: Local filesystem path to an image file.
            url: Publicly accessible URL of an image.

        Raises:
            ValueError: If neither or both of ``path``/``url`` are given.
        """
        if bool(path) == bool(url):
            raise ValueError("Provide exactly one of 'path' or 'url'.")
        if url:
            return {"url": url}
        assert path is not None
        return {"base64": encode_file_to_base64(path), "mime_type": detect_mime_type(path)}

    @staticmethod
    def load_file(path: str) -> FilePayload:
        """Build a :class:`FilePayload` from a local file path."""
        import os

        return {
            "base64": encode_file_to_base64(path),
            "mime_type": detect_mime_type(path),
            "name": os.path.basename(path),
        }

    def _should_retry_exception(self, exc: BaseException) -> bool:
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, TeaLowError) and is_retryable_status(exc.status_code):
            return True
        from .exceptions import ConnectionError as TeaLowConnectionError
        from .exceptions import TimeoutError as TeaLowTimeoutError

        return isinstance(exc, (TeaLowConnectionError, TeaLowTimeoutError))

    def send(
        self,
        prompt: str,
        *,
        images: Optional[List[ImagePayload]] = None,
        files: Optional[List[FilePayload]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        remember: bool = True,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Send a message to the configured model and return its response.

        Args:
            prompt: The user's message text.
            images: Optional list of images for vision-capable models.
                Build entries with :meth:`load_image`.
            files: Optional list of non-image file attachments. Build
                entries with :meth:`load_file`.
            temperature: Overrides the client's default temperature.
            max_tokens: Overrides the client's default max_tokens.
            json_mode: If True, instructs the provider to return valid
                JSON and validates that the response parses as JSON.
            remember: Whether to record this exchange in conversation history.
            extra_params: Additional provider-specific parameters merged
                directly into the outgoing request body.

        Returns:
            A :class:`Response` object. ``str(response)`` yields the text.

        Raises:
            TeaLowError: Or one of its subclasses, depending on the failure.
        """
        self.conversation.add_user_message(prompt, images=images, files=files)
        try:
            response = retry_sync(
                lambda: self._provider.send(
                    self.conversation,
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                    json_mode=json_mode,
                    extra_params=extra_params,
                ),
                max_retries=self.max_retries,
                retry_on=(TeaLowError,),
                should_retry=self._should_retry_exception,
                on_retry=lambda attempt, exc, delay: self._logger.warning(
                    "Retry %d/%d after error: %s (sleeping %.2fs)",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                ),
                provider=self.provider,
            )
        except RetryExhaustedError:
            if not remember:
                self.conversation.messages.pop()
            raise
        except TeaLowError:
            if not remember:
                self.conversation.messages.pop()
            raise

        if remember:
            self.conversation.add_assistant_message(response.text)
        else:
            self.conversation.messages.pop()
        return response

    def stream(
        self,
        prompt: str,
        *,
        images: Optional[List[ImagePayload]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        remember: bool = True,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[StreamChunk]:
        """Stream a response from the model incrementally.

        Yields :class:`StreamChunk` objects as they arrive. After
        iteration completes, if ``remember`` is True, the fully
        assembled assistant message is appended to conversation history.

        Example:
            >>> for chunk in ai.stream("Tell me a story"):
            ...     print(chunk.delta, end="", flush=True)
        """
        self.conversation.add_user_message(prompt, images=images)
        accumulator = StreamAccumulator(model=self.model, provider=self.provider)

        def _do_stream() -> Iterator[StreamChunk]:
            return self._provider.stream(
                self.conversation,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                extra_params=extra_params,
            )

        try:
            for chunk in accumulator.consume(_do_stream()):
                yield chunk
        except TeaLowError:
            if not remember:
                self.conversation.messages.pop()
            raise

        final = accumulator.finalize()
        if remember:
            self.conversation.add_assistant_message(final.text)
        else:
            self.conversation.messages.pop()

    def generate_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate one or more images from a text prompt.

        Returns:
            A list of image URLs (OpenAI) or Base64-encoded image data
            (Gemini/Imagen), depending on the provider.

        Raises:
            ImageGenerationError: If the configured provider/model does
                not support image generation.
        """
        return retry_sync(
            lambda: self._provider.generate_image(
                prompt, size=size, n=n, extra_params=extra_params
            ),
            max_retries=self.max_retries,
            retry_on=(TeaLowError,),
            should_retry=self._should_retry_exception,
            provider=self.provider,
        )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._session.close()

    def __enter__(self) -> "TeaLow":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TeaLow(model={self.model!r}, provider={self.provider!r})"
