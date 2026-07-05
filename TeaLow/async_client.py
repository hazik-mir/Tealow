"""Asynchronous client for the TeaLow unified AI SDK.

Mirrors :class:`TeaLow.client.TeaLow` but exposes ``async``/``await``
compatible methods, suitable for use inside asyncio event loops (e.g.
web servers, bots, or concurrent batch pipelines).

Example:
    >>> import asyncio
    >>> from TeaLow import AsyncTeaLow
    >>>
    >>> async def main():
    ...     ai = AsyncTeaLow(model="gpt-4o-mini", api="YOUR_API_KEY")
    ...     response = await ai.send("Hello!")
    ...     print(response)
    ...     await ai.close()
    >>>
    >>> asyncio.run(main())
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from .client import detect_provider
from .constants import (
    DEFAULT_BASE_URLS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)
from .exceptions import RateLimitError, TeaLowError
from .models import Conversation, Response, StreamChunk
from .providers import get_provider_class
from .providers.base import BaseProvider
from .session import AsyncHTTPSession, HTTPSession
from .streaming import StreamAccumulator
from .types import ImagePayload
from .utils import (
    get_logger,
    is_retryable_status,
    resolve_api_key,
    retry_async,
    validate_api_key_format,
)


class AsyncTeaLow:
    """Unified asynchronous client for interacting with multiple AI providers.

    See :class:`TeaLow.client.TeaLow` for full parameter documentation;
    the constructor signature and semantics are identical, only the
    request methods (:meth:`send`, :meth:`stream`) are coroutines.
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

        self._logger = get_logger("async_client")
        self._async_session = AsyncHTTPSession(
            timeout=timeout,
            proxies=proxies,
            custom_headers=headers,
            verify_ssl=verify_ssl,
        )
        # A synchronous session is kept alongside the async one purely to
        # back image-generation calls, which providers implement as a
        # single blocking HTTP call executed in a background thread (see
        # generate_image below) rather than duplicating that logic async.
        self._sync_session_for_images = HTTPSession(
            timeout=timeout,
            proxies=proxies,
            custom_headers=headers,
            verify_ssl=verify_ssl,
        )

        provider_class = get_provider_class(self.provider_enum)
        self._provider: BaseProvider = provider_class(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            session=self._sync_session_for_images,
            async_session=self._async_session,
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

    def _should_retry_exception(self, exc: BaseException) -> bool:
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, TeaLowError) and is_retryable_status(exc.status_code):
            return True
        from .exceptions import ConnectionError as TeaLowConnectionError
        from .exceptions import TimeoutError as TeaLowTimeoutError

        return isinstance(exc, (TeaLowConnectionError, TeaLowTimeoutError))

    async def send(
        self,
        prompt: str,
        *,
        images: Optional[List[ImagePayload]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        remember: bool = True,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Asynchronously send a message and return the model's response."""
        self.conversation.add_user_message(prompt, images=images)

        async def _do_send() -> Response:
            return await self._provider.asend(
                self.conversation,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                json_mode=json_mode,
                extra_params=extra_params,
            )

        try:
            response = await retry_async(
                _do_send,
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
        except TeaLowError:
            if not remember:
                self.conversation.messages.pop()
            raise

        if remember:
            self.conversation.add_assistant_message(response.text)
        else:
            self.conversation.messages.pop()
        return response

    async def stream(
        self,
        prompt: str,
        *,
        images: Optional[List[ImagePayload]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        remember: bool = True,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronously stream a response, yielding chunks as they arrive."""
        self.conversation.add_user_message(prompt, images=images)
        accumulator = StreamAccumulator(model=self.model, provider=self.provider)

        raw_stream = self._provider.astream(
            self.conversation,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            extra_params=extra_params,
        )

        try:
            async for chunk in accumulator.aconsume(raw_stream):
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

    async def generate_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Asynchronously generate one or more images from a text prompt.

        Note:
            Provider adapters implement image generation synchronously
            (it is a single non-streaming HTTP call); this coroutine
            runs it in a background thread so it does not block the
            event loop.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._provider.generate_image(
                prompt, size=size, n=n, extra_params=extra_params
            ),
        )

    async def close(self) -> None:
        """Release the underlying async and sync HTTP clients."""
        await self._async_session.close()
        self._sync_session_for_images.close()

    async def __aenter__(self) -> "AsyncTeaLow":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"AsyncTeaLow(model={self.model!r}, provider={self.provider!r})"
